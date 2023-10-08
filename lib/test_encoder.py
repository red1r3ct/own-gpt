from lib.encoders import byte_pair_encoder, byte_pair_decoder

def test_bp_encoder():
    sample_text = ['Some sample text!']
    encoded = byte_pair_encoder(sample_text)
    decoded = byte_pair_decoder(encoded)

    assert len(encoded[0]) < len(sample_text[0])
    assert decoded[0] == sample_text[0]
