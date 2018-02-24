def cuda_check(module_list):
    cuda = False
    for mod in module_list:
        cuda = mod.is_cuda
        if cuda:
            break
    if not cuda:
        return module_list

    module_list_new = []
    for mod in module_list:
        module_list_new.append(mod.cuda())
    return module_list_new
