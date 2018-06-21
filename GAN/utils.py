def cuda_check(module_list):
    """
    Checks if any module or variable in a list has cuda() true and if so
    moves complete list to cuda

    Parameters
    ----------
    module_list : list
        List of modules/variables

    Returns
    -------
    module_list_new : list
        Modules from module_list all moved to the same device
    """
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
