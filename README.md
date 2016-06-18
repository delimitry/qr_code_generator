# QR Code generator
QR Code generator in Python


## Dependencies:  
PIL - Python Imaging Library (or Pillow)


## Usage:
```python
qr_code = QRCode(error_correction_level='L', mode_name='byte', mask_number=2, data='encoded message')
qr_code.save('qr.png', module_size=5)
```

This will generate an image file "qr.png" with a QR Code that contains "encoded message" text.

![QR](https://3.bp.blogspot.com/-plghUd7OfTs/V2V2IdyOfBI/AAAAAAAAAwI/6KtAgtRr1gkwHHgHcGUE59Eh5r8Uj7EvACLcB/s1600/qr_encoded_message.png "QR with encoded message")


## License:
Released under [The MIT License](https://github.com/delimitry/qr_code_generator/blob/master/LICENSE).

The word "QR Code" is registered trademark of DENSO WAVE INCORPORATED  
http://www.denso-wave.com/qrcode/faqpatent-e.html
