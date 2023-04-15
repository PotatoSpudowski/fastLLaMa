import sysconfig
import platform

lib_path: str = sysconfig.get_path('stdlib')
include_path: str = sysconfig.get_path('include')
version: str = platform.python_version()
print(f'{lib_path};{include_path};{version}', end='')