import json

class Arg:
  def __init__(self, **entries):
    self.__dict__.update(entries)

class ModelConfig:
  def __init__(self, config_path):
    self.config_path = config_path
    f = open(self.config_path, 'r')
    self.config_json = json.load(f)
    self.arg = Arg(**self.config_json)

  def get_config(self):
    return self.arg

class ElectraConfig(ModelConfig):
  def __init__(self,config_path = '../pretrain/config/electra-train.json'):
    super().__init__(config_path)

  def get_config(self):
    paths = [self.arg.generator_config_path, self.arg.discriminator_config_path]

    def arg(path):
      f = open(path, 'r')
      config_json = json.load(f)
      return Arg(**config_json)

    configs = list(map(arg,paths))

    return self.arg, configs[0], configs[1] # train_args, gen_args,

if __name__=='__main__':
  configs = ElectraConfig().get_config()
  print(configs)


