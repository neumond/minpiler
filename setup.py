from pathlib import Path
from setuptools import setup


longdesc = (Path(__file__).parent / 'README.md').read_text()


setup(
    name='minpiler',
    version='0.2.0',
    description='Compile python code to Mindustry microprocessor instructions',
    long_description=longdesc,
    long_description_content_type='text/markdown',
    url='https://github.com/neumond/minpiler',
    author='Vitalik Verhovodov',
    author_email='knifeslaughter@gmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Education',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Topic :: Games/Entertainment',
    ],
    keywords='mindustry',
    packages=['minpiler'],
    install_requires=[],
    entry_points={
        'console_scripts': ['minpiler = minpiler.cmdline:run'],
    },
)
