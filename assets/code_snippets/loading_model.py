state_dict = torch.load(model_path)
ModelClass = SmartSequential.module_dict[state_dict["MODEL_CLASS"]]    
model = ModelClass(state_dict["CONFIG"])            # set the architecture
model.load_state_dict(state_dict)                   # load the params