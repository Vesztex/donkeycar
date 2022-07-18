import glob
import os
import sys
import types
import logging
import importlib.util

logger = logging.getLogger(__name__)


class Config:
    
    def from_pyfile(self, filename):
        d = types.ModuleType('config')
        d.__file__ = filename
        try:
            with open(filename, mode='rb') as config_file:
                exec(compile(config_file.read(), filename, 'exec'), d.__dict__)
        except IOError as e:
            e.strerror = 'Unable to load configuration file (%s)' % e.strerror
            raise
        self.from_object(d)
        return True

    def from_object(self, obj):
        for key in dir(obj):
            if key.isupper():
                setattr(self, key, getattr(obj, key))

    def __str__(self):
        result = []
        for key in dir(self):
            if key.isupper():
                result.append((key, getattr(self, key)))
        return str(result)

    def show(self):
        for attr in dir(self):
            if attr.isupper():
                print(attr, ":", getattr(self, attr))


def load_config(config_path=None, myconfig="myconfig.py"):
    
    if config_path is None:
        import __main__ as main
        main_path = os.path.dirname(os.path.realpath(main.__file__))
        config_path = os.path.join(main_path, 'config.py')
        if not os.path.exists(config_path):
            local_config = os.path.join(os.path.curdir, 'config.py')
            if os.path.exists(local_config):
                config_path = local_config

    logger.info(f'Loading config file: {config_path}')
    cfg = Config()
    cfg.from_pyfile(config_path)

    # look for the optional myconfig.py in the same path.
    personal_cfg_path = config_path.replace("config.py", myconfig)
    if os.path.exists(personal_cfg_path):
        logger.info(f"Loading personal config over-rides from {myconfig}")
        personal_cfg = Config()
        personal_cfg.from_pyfile(personal_cfg_path)
        cfg.from_object(personal_cfg)
    else:
        logger.warning(f"Personal config file {myconfig} not found at"
                       f" {personal_cfg_path}")
    # check if myparts path is given and import modules from there
    myparts_path = getattr(cfg, 'MYPARTS_PATH', None)
    if myparts_path:
        file_path = glob.glob(os.path.join(myparts_path, '*.py'))
        for f in file_path:
            try:
                module_name = os.path.splitext(os.path.basename(f))[0]
                spec = importlib.util.spec_from_file_location(module_name, f)
                module = importlib.util.module_from_spec(spec)
                assert module_name not in sys.modules, \
                    f"Module {module_name} already loaded, skipping..."
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                logger.info(f'Imported module {module_name}')
            except Exception as e:
                logger.error(e)
    return cfg
