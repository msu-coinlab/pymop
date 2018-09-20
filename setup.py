from setuptools import setup, find_packages



def readme():
    with open('README.rst') as f:
        return f.read()


setup(
    name="pymop",
    version="0.2.1",
    author="Julian Blank",
    author_email="blankjul@egr.msu.edu",
    description="Optimization Test Problems",
    long_description=readme(),
    url="https://github.com/msu-coinlab/pymop",
    license='MIT',
    keywords="optimization",
    install_requires=['numpy'],
    packages=find_packages(exclude=['tests', 'docs']),
    include_package_data=True,
)
