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

import json
import os

import tensorflow as tf
# from tf2_net import Net
from myia_model import Myia
from tensorflow.keras import layers, models
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import ModelLearnable, make_model_learnable
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_common.app_constant import AppConstants
from nvflare.fuel.utils import fobs


class TF2ModelPersistor(ModelPersistor):
    def __init__(self, save_name = "tf2 _model.fobs"):
        super().__init__()
        self.save_name = save_name

    def _initialize(self, fl_ctx: FLContext):
        # get save path from FLContext
        app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
        env = None
        run_args = fl_ctx.get_prop(FLContextKey.ARGS)
        if run_args:
            env_config_file_name = os.path.join(app_root, run_args.env)
            if os.path.exists(env_config_file_name):
                try:
                    with open(env_config_file_name) as file:
                        env = json.load(file)
                except:
                    self.system_panic(
                        reason="error opening env config file {}".format(env_config_file_name), fl_ctx=fl_ctx
                    )
                    return

        if env is not None:
            if env.get("APP_CKPT_DIR", None):
                fl_ctx.set_prop(AppConstants.LOG_DIR, env["APP_CKPT_DIR"], private=True, sticky=True)
            if env.get("APP_CKPT") is not None:
                fl_ctx.set_prop(
                    AppConstants.CKPT_PRELOAD_PATH,
                    env["APP_CKPT"],
                    private=True,
                    sticky=True,
                )

        log_dir = fl_ctx.get_prop(AppConstants.LOG_DIR)
        if log_dir:
            self.log_dir = os.path.join(app_root, log_dir)
        else:
            self.log_dir = app_root
        self._fobs_save_path = os.path.join(self.log_dir, self.save_name)
        print(os.path.join(self.log_dir, self.save_name))
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        fl_ctx.sync_sticky()

    def load_model(self, fl_ctx: FLContext) -> ModelLearnable:
        """Initializes and loads the Model.

        Args:
            fl_ctx: FLContext

        Returns:
            Model object
        """

        if os.path.exists(self._fobs_save_path):
            self.logger.info("Loading server weights")
            with open(self._fobs_save_path, "rb") as f:
                model_learnable = fobs.load(f)
        else:
            self.logger.info("Initializing server model")
            # from myiatrainer.py
            model = models.Sequential([
            # model trains if below line input_shape=(150, 200, 3) ????, rbg and alpha layer?
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 200, 3)),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                # removed the extra dense layers and leave for one
                layers.Dense(64, activation='relu'),
                # *[layers.Dense(64, activation='relu') for _ in range(1)],  # Add dense layers based on config
                layers.Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            _ = model(tf.keras.Input(shape=(150, 200, 3)))
            var_dict = self.get_the_layer(model)
            model_learnable = make_model_learnable(var_dict, dict())
        return model_learnable

    def get_the_layer(self, model):
        layers = {}
        # var_dict = {model.get_layer(index=key).name: value for key, value in enumerate(model.weights)}
        for key, value in enumerate(model.weights):
            if(key < 7):
                layers[model.get_layer(index=key).name] = value
                # print(key)
                # print(value)
                # print(model.get_layer(index=key).name)
        
        return layers

    def handle_event(self, event: str, fl_ctx: FLContext):
        if event == EventType.START_RUN:
            self._initialize(fl_ctx)

    def save_model(self, model_learnable: ModelLearnable, fl_ctx: FLContext):
        """Saves model.

        Args:
            model_learnable: ModelLearnable object
            fl_ctx: FLContext
        """
        model_learnable_info = {k: str(type(v)) for k, v in model_learnable.items()}
        self.logger.info(f"Saving aggregated server weights: \n {model_learnable_info}")
        with open(self._fobs_save_path, "wb") as f:
            fobs.dump(model_learnable, f)
