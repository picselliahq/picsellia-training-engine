with strategy.scope():
      model_lib_v2.train_loop(
          pipeline_config_path="project 21/fastrcnn/0/config/",
          model_dir="project 21/fastrcnn/0/exported_model/",
          train_steps=500,
          use_tpu=False,
          checkpoint_every_n=200,
          record_summaries=True)