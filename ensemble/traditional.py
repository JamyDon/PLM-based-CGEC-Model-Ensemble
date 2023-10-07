def get_models(path):
    models = []
    with open(path, "r") as f:
        for line in f:
            if line[0] == "#" or line[0] == "\n":
                continue
            models.append(line.rstrip("\n"))
    return models
