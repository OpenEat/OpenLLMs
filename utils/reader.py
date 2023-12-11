import yaml


def read_yaml(file):
    """ read_yaml """
    with open(file, "r", encoding="utf-8") as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)
    return config

def read_text(file):
    """ read_text """
    lines = []
    with open(file) as f:
        for line in f:
            line = line.strip()
            lines.append(line)
    return lines