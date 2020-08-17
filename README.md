# Training_ANN_with_Pytorch
 This repo aims to train neural networks with pytorch. The data are normalized insede the class. The validation is performed automatically. The output model gets the unscaled data and returns the output unscaled (The scaling is performed inside). This way the user do not interact the scaling,; however the user can select to not use the default scaling and scale the data before the training


# Foobar

Foobar is a Python library for dealing with word pluralization.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install foobar
```

## Usage

```python
import foobar

foobar.pluralize('word') # returns 'words'
foobar.pluralize('goose') # returns 'geese'
foobar.singularize('phenomena') # returns 'phenomenon'
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
