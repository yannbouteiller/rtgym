from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(name='rtgym',
      packages=[package for package in find_packages()],
      version='0.4',
      license='MIT',
      description='Easily implement custom OpenAI Gym environments for real-time applications',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='Yann Bouteiller',
      url='https://github.com/yannbouteiller/rtgym',
      download_url='https://github.com/yannbouteiller/rtgym/archive/v0.4.tar.gz',
      keywords=['gym', 'real', 'time', 'custom', 'environment', 'reinforcement', 'learning', 'random', 'delays'],
      install_requires=['gym', 'numpy'],
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Information Technology',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX',
          'Programming Language :: Python',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
      )
