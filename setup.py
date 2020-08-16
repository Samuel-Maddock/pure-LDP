from setuptools import setup, find_packages
import shutil

with open("README.md", "r") as fh:
    long_description = fh.read()

shutil.rmtree("./build")
shutil.rmtree("./dist")
shutil.rmtree("./pure_ldp.egg-info")

setup(
    name='pure-ldp',
    version='1.0.5',
    packages=find_packages(exclude=['*development*', "*apple_sf*", "*treehistogram*", "*rappor*",
                                    "*priv_count_sketch*", "*hashtogram*", "*explicit_hist*", ]),
    install_requires=["xxhash", "numpy", "scipy", "bitstring", "bitarray"],
    url='https://github.com/Samuel-Maddock/pure-LDP',
    license='MIT',
    author='Samuel Maddock',
    author_email='samuel-maddock@hotmail.com',
    description='Simple pure LDP frequency oracle implementations',
    long_description=long_description,
    long_description_content_type="text/markdown",
)
