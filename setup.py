from setuptools import setup, find_packages

    
setup(name='eBO',
    version=0.0,
    description='eBO',
    classifiers=[
    'Development Status :: 2 - Pre-Alpha',
    'Programming Language :: Python :: 3.11.1',
    'Topic :: BO for EPICS'
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
