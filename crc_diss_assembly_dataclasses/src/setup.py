# setup.py

import setuptools


setuptools.setup(
    name='diss_assembly_dataclasses',
    version='1.0',
    author='Malte Hansjosten',
    description='Dataclasses for representing assemblies in the context of disassembly planning.',
    long_description='Dataclasses for representing assemblies in the context of disassembly planning.',
    license='MIT',
    install_requires=['numpy', 'networkx', 'lxml','open3d'],
    packages=setuptools.find_packages(),

)
