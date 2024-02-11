import yaml
import paramgen
from pprint import pprint


class DAGParams:
    def __init__(self, params_file_path: str=None) -> None:
        if params_file_path:
            self.load_params(params_file_path)
        else:
            self.params = {}

    def load_params(self, params_file_path: str):
        with open(params_file_path, "r") as f:
            self.params = yaml.safe_load(f)

    def save_params(self, params_file_path: str):
        with open(params_file_path, "w") as f:
            yaml.dump(self.params, f)

    def set_params(self, params: dict):
        self.params = params


if __name__ == "__main__":
    params = DAGParams()
    generator = paramgen.DAGParamGenerator()
    params.set_params(generator.generate_param())
    params.save_params("temp_params.yaml")
    loaded_params = DAGParams("temp_params.yaml")
    assert params.params == loaded_params.params
    pprint(params.params)
    pprint(loaded_params.params)
