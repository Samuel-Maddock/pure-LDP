from setuptools import setup, find_packages

setup(
    name='pure-ldp',
    version='1.0.0',
    packages=find_packages(),
    install_requires=["xxhash", "numpy", "scipy"],
    url='https://github.com/Samuel-Maddock/pure-LDP',
    license='',
    author='Samuel Maddock',
    author_email='samuel-maddock@hotmail.com',
    description='Simple pure LDP frequency oracle implementations'
)
