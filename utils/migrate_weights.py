def migrate_weights(single_model, complex_model):
    
    state_dict = {}
    single_dict = single_model.state_dict()
    

    for name in ['token_embedding.weight', 'output_proj.weight', 'output_norm.weight']:
        state_dict[name] = single_dict[name]
    

    state_dict['time_embedding.pe'] = single_dict['time_embedding.pe']
    state_dict['time_embedding.scale_proj.weight'] = single_dict['time_embedding.scale_proj.weight']
    state_dict['time_embedding.scale_proj.bias'] = single_dict['time_embedding.scale_proj.bias']
    state_dict['time_embedding.bias_proj.weight'] = single_dict['time_embedding.bias_proj.weight']
    state_dict['time_embedding.bias_proj.bias'] = single_dict['time_embedding.bias_proj.bias']
    

    state_dict['position_embedding_pep.positional_embedding.weight'] = \
        single_dict['position_embedding.positional_embedding.weight']

    num_layers = complex_model.num_layers
    for i in range(num_layers):

        prefix = f'layers.{i}.self_attention.attn.'
        single_prefix = f'layers.{i}.attention.attn.'
        state_dict[prefix + 'in_proj_weight'] = single_dict[single_prefix + 'in_proj_weight']
        state_dict[prefix + 'in_proj_bias'] = single_dict[single_prefix + 'in_proj_bias']
        state_dict[prefix + 'out_proj.weight'] = single_dict[single_prefix + 'out_proj.weight']
        state_dict[prefix + 'out_proj.bias'] = single_dict[single_prefix + 'out_proj.bias']
        
        ff_prefix = f'layers.{i}.feed_forward.'
        state_dict[ff_prefix + 'gate_proj.weight'] = single_dict[ff_prefix + 'gate_proj.weight']
        state_dict[ff_prefix + 'gate_proj.bias'] = single_dict[ff_prefix + 'gate_proj.bias']
        state_dict[ff_prefix + 'up_proj.weight'] = single_dict[ff_prefix + 'up_proj.weight']
        state_dict[ff_prefix + 'up_proj.bias'] = single_dict[ff_prefix + 'up_proj.bias']
        state_dict[ff_prefix + 'down_proj.weight'] = single_dict[ff_prefix + 'down_proj.weight']
        state_dict[ff_prefix + 'down_proj.bias'] = single_dict[ff_prefix + 'down_proj.bias']
        
        state_dict[f'layers.{i}.norm1.weight'] = single_dict[f'layers.{i}.norm1.weight']
        state_dict[f'layers.{i}.norm3.weight'] = single_dict[f'layers.{i}.norm2.weight'] 
    

    complex_model.load_state_dict(state_dict, strict=False)
    return complex_model