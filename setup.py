from setuptools import setup, find_packages


# Get the long description from the README file
#def readme():
#    with open('README.rst') as f:
#        return f.read()

setup(name='ddgclib',
      version='0.4.3',
      description='Discrete differential geometry curvature library used for mean curvature flow and finite volume method simulations.',
      url='https://github.com/Leibniz-IWT/ddgclib',
      author='Stefan Endres, Lutz Mädler, Ianto Cannon, Sonyi Deng, Marcello Zani',
      author_email='s.endres@iwt-uni-bremen.de',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'scipy',
          'numpy',
          'hyperct',
           ],
      extras_require={
          'vis': ['polyscope', 'matplotlib'],
          'data': ['pandas'],
          'dev': ['pytest', 'pytest-cov'],
      },
      long_description_content_type='text/markdown',
      keywords='navier-stokes, pde, finite volume, mean curvature flow, discrete differential geometry',
      classifiers=[
          'Development Status :: 3 - Alpha',

          'Intended Audience :: Science/Research',
          'Intended Audience :: Developers',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Mathematics',

          'License :: OSI Approved :: MIT License',

          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          'Programming Language :: Python :: 3.11',
          'Programming Language :: Python :: 3.12',
          'Programming Language :: Python :: 3.13',
          'Operating System :: OS Independent',
      ],
      python_requires='>=3.9',
      zip_safe=False)