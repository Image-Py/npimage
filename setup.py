from setuptools import setup

descr = """scipy.ndimage implemented in pure numpy"""

if __name__ == '__main__':
    setup(name='npimage',
        version='0.01',
        url='https://github.com/Image-Py/npimage',
        description='ndimage in numpy',
        long_description=descr,
        author='YXDragon',
        author_email='yxdragon@imagepy.org',
        license='BSD 3-clause',
        packages=['npimage'],
        package_data={},
        install_requires=[
            'numpy'
        ],
    )
