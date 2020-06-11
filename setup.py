from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='pure-ldp',
    version='1.0.3',
    packages=find_packages(),
    install_requires=["xxhash", "numpy", "scipy", "bitstring", "bitarray"],
    url='https://github.com/Samuel-Maddock/pure-LDP',
    license='',
    author='Samuel Maddock',
    author_email='samuel-maddock@hotmail.com',
    description='Simple pure LDP frequency oracle implementations',
    long_description=long_description,
    long_description_content_type="text/markdown",
)
