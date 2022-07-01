from setuptools import setup, find_packages


setup(
    name='glaucoma',
    version='0.1',
    description='Glaucoma machine learning code',
    author='Felipe Marcelino',
    author_email='felipe.marcelino1991@gmai.com',
    install_requires=['wheel'], #external packages as dependencies
    scripts=[],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.6"
)
