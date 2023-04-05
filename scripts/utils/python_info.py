import sysconfig

lib_path: str = sysconfig.get_path('stdlib')
include_path: str = sysconfig.get_path('include')
print(f'{lib_path};{include_path}', end='')