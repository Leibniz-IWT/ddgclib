from setuptools import setup, find_packages


# Get the long description from the README file
#def readme():
#    with open('README.rst') as f:
#        return f.read()

setup(name='ddgclib',
      version='0.4.2',
      description='Discrete differential geometry curvature library used for mean curvature flow and finite volume method simulations.',
      #url='https://github.com/stefan-endres/',
      author='Stefan Endres, Lutz Mädler, Sonyi Deng, Marcello Zani',
      author_email='s.endres@iwt-uni-bremen.de',
      license='MIT',
      packages=['ddgclib'],
      install_requires=[
          'scipy',
          'numpy',
          'hyperct'
          'polyscope'
         # 'pytest',
         # 'pytest-cov'
           ],
      #long_description=readme(),
      long_description='None',
      long_description_content_type='text/markdown',
      keywords='optimization, navier-stokes, pde, finite volume, mean curvature flow, discrete differential geometry',
      classifiers=[
          # How mature is this project? Common values are
          #   3 - Alpha
          #   4 - Beta
          #   5 - Production/Stable
          'Development Status :: 3 - Alpha',

          'Intended Audience :: Science/Research',
          'Intended Audience :: Developers',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Mathematics',

          'License :: OSI Approved :: MIT License',

          'Programming Language :: Python :: 3',
          'Operating System :: OS Independent',
      ],
      #test_suite='shgo.tests.test__shgo',  #TODO
      python_requires='>=3.9',
      zip_safe=False)