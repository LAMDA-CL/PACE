def get_model(model_name, args):
    name = model_name.lower()
    if name=="pace":
        from models.pace import Learner
        return Learner(args)
    else:
        assert 0
