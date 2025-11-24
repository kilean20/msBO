from setuptools import setup, find_packages

    
setup(name='boom',
    version=0.0,
    description='Bayesian Optimization On a Machine',
    classifiers=[
    'Development Status :: 2 - Pre-Alpha',
    'Programming Language :: Python :: 3.11.1',
    ],
    keywords = ['EPICS','BO','multistate-BO','multitask-BO','machine-learning', 'optimization', 'FRIB beam tuning'],
    author='Kilean Hwang',
    author_email='hwang@frib.msu.edu',
#     license='',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'pytorch',
        'gpytorch',
        'botorch',
    ],
    zip_safe=False)
