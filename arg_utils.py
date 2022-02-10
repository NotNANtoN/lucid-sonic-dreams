from hydra import initialize, compose
import hydra

def is_notebook():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False

def get_cfg():
    args = {}
    @hydra.main(config_path="conf", config_name="config")
    def command_line_cfg(cfg):
        cfg = dict(cfg)
        args.update(cfg)

    if is_notebook():
        with initialize(config_path="conf/"):
            args = compose(config_name="config.yaml", overrides=[])
    else:
        command_line_cfg()
    return dict(args)
