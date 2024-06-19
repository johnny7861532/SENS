from setuptools import setup, find_packages

setup(
    name='SENS',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'transformers',
        'numpy',
        'pandas',
        'scikit-learn',
        'joblib'
    ],
    author='Johnny_Hsieh',
    author_email='johnny@morphusai.com',
    description='SENS is a proprietary emotion understanding module of MorphusAI, specifically designed for processing Chinese conversations. It enables our persona models to deeply understand emotions in conversations and respond more humanely.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/johnny7861532/SENS',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
