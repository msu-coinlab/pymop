from setuptools import setup, find_packages


def readme():
    with open('README.rst') as f:
        return f.read()


__name__ = "pymop"
__author__ = "Julian Blank"
__version__ = '0.2.4.dev'
__url__ = "https://github.com/msu-coinlab/pymop"

setup(
    name=__name__,
    version=__version__,
    author=__author__,
    author_email="blankjul@egr.msu.edu",
    description="Multi-Objective Optimization Problems",
    long_description=readme(),
    url=__url__,
    license='Apache License 2.0',
    keywords="optimization",
    install_requires=['numpy', "autograd"],
    packages=find_packages(exclude=['tests', 'docs']),
    include_package_data=True,
    platforms='any',
)
