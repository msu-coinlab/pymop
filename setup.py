from setuptools import setup


setup(
    name="pyop",
    version="0.0.1",
    author="Julian Blank",
    description="Optimization Problems",
    license='MIT',
    keywords="optimization",
    packages=['pyop', 'pyop/problems'],
    install_requires=['numpy']
)
