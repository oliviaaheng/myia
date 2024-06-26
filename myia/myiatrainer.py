# took from v=nvflare example cifar10trainer.py

# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os.path
# myia specific
from model_builder import create_model

import torch
from pt_constants import PTConstants 
# from simple_network import SimpleNetwork
from torch import nn
from torch.optim import SGD
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.model import make_model_learnable, model_learnable_to_dxo
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_opt.pt.model_persistence_format_manager import PTModelPersistenceFormatManager


class MyiaTrainer(Executor):
    def __init__(
        self,
        # myia specific
        epochs=10,
        layers = 2,
        model_name = "test_model",
        # nvflare specific
        train_task_name=AppConstants.TASK_TRAIN,
        submit_model_task_name=AppConstants.TASK_SUBMIT_MODEL,
        exclude_vars=None,
        pre_train_task_name=AppConstants.TASK_GET_WEIGHTS,
    
    ):
        """Cifar10 Trainer handles train and submit_model tasks. During train_task, it trains a
        simple network on CIFAR10 dataset. For submit_model task, it sends the locally trained model
        (if present) to the server.

        Args:
            lr (float, optional): Learning rate. Defaults to 0.01
            epochs (int, optional): Epochs. Defaults to 5
            train_task_name (str, optional): Task name for train task. Defaults to "train".
            submit_model_task_name (str, optional): Task name for submit model. Defaults to "submit_model".
            exclude_vars (list): List of variables to exclude during model loading.
            pre_train_task_name: Task name for pre train task, i.e., sending initial model weights.
        """
        super().__init__()

        self._epochs = epochs
        self._layers = layers
        self._model_name = model_name
        self._train_task_name = train_task_name
        self._pre_train_task_name = pre_train_task_name
        self._submit_model_task_name = submit_model_task_name
        self._exclude_vars = exclude_vars

        # Training setup
        self.model = create_model(label_good_dir, label_bad_dir, {'epochs': epochs,  'no_layers': layers, 'model_name': model_name})
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # self._train_dataset = CIFAR10(root=data_path, transform=transforms, download=True, train=True)
        # self._train_loader = DataLoader(self._train_dataset, batch_size=4, shuffle=True)
        # self._n_iterations = len(self._train_loader)
        # TODO: change from hardcoded
        self._n_iterations = 13

        # # Setup the persistence manager to save PT model.
        # # The default training configuration is used by persistence manager
        # # in case no initial model is found.
        self._default_train_conf = {"train": {"model": type(self.model).__name__}}
        self.persistence_manager = PTModelPersistenceFormatManager(
            data=self.model.state_dict(), default_train_conf=self._default_train_conf
        )

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        try:
            if task_name == self._pre_train_task_name:
                # Get the new state dict and send as weights
                return self._get_model_weights()
            elif task_name == self._train_task_name:
                # Get model weights
                print("worked in execute")
                try:
                    dxo = from_shareable(shareable)
                except:
                    self.log_error(fl_ctx, "Unable to extract dxo from shareable.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Ensure data kind is weights.
                if not dxo.data_kind == DataKind.WEIGHTS:
                    self.log_error(fl_ctx, f"data_kind expected WEIGHTS but got {dxo.data_kind} instead.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Convert weights to tensor. Run training
                torch_weights = {k: torch.as_tensor(v) for k, v in dxo.data.items()}
                self._local_train(fl_ctx, torch_weights, abort_signal)

                # Check the abort_signal after training.
                # local_train returns early if abort_signal is triggered.
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)

                # Save the local model after training.
                self._save_local_model(fl_ctx)

                # Get the new state dict and send as weights
                return self._get_model_weights()
            elif task_name == self._submit_model_task_name:
                # Load local model
                ml = self._load_local_model(fl_ctx)

                # Get the model parameters and create dxo from it
                dxo = model_learnable_to_dxo(ml)
                return dxo.to_shareable()
            else:
                return make_reply(ReturnCode.TASK_UNKNOWN)
        except Exception as e:
            self.log_exception(fl_ctx, f"Exception in simple trainer: {e}.")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)


# check the logs in the good example and see what is different between ours and theirs
# see what nvida is doing and calling before the train
               

    def _get_model_weights(self) -> Shareable:
        # Get the new state dict and send as weights
        weights = {k: v.cpu().numpy() for k, v in self.model.state_dict().items()}

        outgoing_dxo = DXO(
            data_kind=DataKind.WEIGHTS, data=weights, meta={MetaKey.NUM_STEPS_CURRENT_ROUND: self._n_iterations}
        )
        return outgoing_dxo.to_shareable()

    def _local_train(self, fl_ctx, weights, abort_signal):
        # Set the model weights
        self.model.load_state_dict(state_dict=weights)

        try:
            # Check directory existence
            if not os.path.exists(train_good_dir) or not os.path.exists(train_bad_dir):
                raise Exception("Training directories do not exist")

            # Load and preprocess images
            train_images = []
            train_labels = []
        

            for filename in os.listdir(train_good_dir):
                if not filename.lower().endswith(('.png', '.jpg')):
                    continue
                try:
                    img = Image.open(os.path.join(train_good_dir, filename))
                    img = img.resize((200, 150)) 
                    img = np.array(img) / 255.0
                    train_images.append(img)
                    train_labels.append(1)  # Label 'good' images as 1
                except Exception as e:
                    print(f"Error processing file {filename}: {str(e)}")

            for filename in os.listdir(train_bad_dir):
                if not filename.lower().endswith(('.png', '.jpg')):
                    continue
                try:
                    img = Image.open(os.path.join(train_bad_dir, filename))
                    img = img.resize((200, 150))
                    img = np.array(img) / 255.0
                    train_images.append(img)
                    train_labels.append(0)  # Label 'bad' images as 0
                except Exception as e:
                    print(f"Error processing file {filename}: {str(e)}")

            # Check if any images were loaded
            if not train_images:
                raise Exception("No images found in training directories")

            # Convert to NumPy arrays
            train_images = np.array(train_images)
            train_labels = np.array(train_labels)
                
            print("in model_builder.py, in create model line 66, before model = ")
            # Define the model architecture
            model = models.Sequential([
                # model trains if below line input_shape=(150, 200, 3) ????, rbg and alpha layer?
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 200, 3)),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                *[layers.Dense(64, activation='relu') for _ in range(config['no_layers'] - 1)],  # Add dense layers based on config
                layers.Dense(1, activation='sigmoid')
            ])
            print("in model_builder.py, in create model line 77, after model = ")
            
            print("in model_builder.py, in create model line 79, before model.compile")
            # Compile the model
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            print("in model_builder.py, in create model line 82, after model.compile")


            print("in model_builder.py, in create model line 85, before model.fit")
            # Train the model
            model.fit(train_images, train_labels, epochs=config['epochs'])
            print("in model_builder.py, in create model line 88, after model.fit")

            model_name = generate_version_name(f"{config['model_name']}.keras", "model/image_model")
            model_path = f"model/image_model/{model_name}"

            # Save the model
            model.save(model_path)

            return {"model_name": model_name, "model_path": model_path}

        except Exception as e:
            print(f"Error creating model: {str(e)}")
            return False

    def _save_local_model(self, fl_ctx: FLContext):
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_prop(ReservedKey.RUN_NUM))
        models_dir = os.path.join(run_dir, PTConstants.PTModelsDir)
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        model_path = os.path.join(models_dir, PTConstants.PTLocalModelName)

        ml = make_model_learnable(self.model.state_dict(), {})
        self.persistence_manager.update(ml)
        torch.save(self.persistence_manager.to_persistence_dict(), model_path)

    def _load_local_model(self, fl_ctx: FLContext):
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_prop(ReservedKey.RUN_NUM))
        models_dir = os.path.join(run_dir, PTConstants.PTModelsDir)
        if not os.path.exists(models_dir):
            return None
        model_path = os.path.join(models_dir, PTConstants.PTLocalModelName)

        self.persistence_manager = PTModelPersistenceFormatManager(
            data=torch.load(model_path), default_train_conf=self._default_train_conf
        )
        ml = self.persistence_manager.to_model_learnable(exclude_vars=self._exclude_vars)
        return ml
