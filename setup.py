from setuptools import setup, find_packages


# Get the long description from the README file
#def readme():
#    with open('README.rst') as f:
#        return f.read()

setup(name='ddgclib',
      version='0.1.0',
      description='Discrete differential goemetry curvature librrayr',
      #url='https://github.com/stefan-endres/',
      author='Stefan Endres, Lutz Mädler',
      author_email='s.endres@iwt-uni-bremen.de',
      license='MIT',
      packages=['ddgclib'],
      install_requires=[
          'scipy',
          'numpy',
         # 'pytest',
         # 'pytest-cov'
           ],
      #long_description=readme(),
      long_description='None',
      long_description_content_type='text/markdown',
      keywords='optimization',
      classifiers=[
          # How mature is this project? Common values are
          #   3 - Alpha
          #   4 - Beta
          #   5 - Production/Stable
          'Development Status :: 3 - Alpha',

          # Indicate who your project is intended for
          'Intended Audience :: Science/Research',
          'Intended Audience :: Developers',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Mathematics',

          # Pick your license as you wish (should match "license" above)
          'License :: OSI Approved :: MIT License',

          # Specify the Python versions you support here. In particular, ensure
          # that you indicate whether you support Python 2, Python 3 or both.
          'Programming Language :: Python :: 3.9',
      ],
      #test_suite='shgo.tests.test__shgo',  #TODO
      zip_safe=False)