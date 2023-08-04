import argparse

def get_parser():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--admin_type', type = str, default = 'dermal', help = 'type of administration. dermal, inhalation, oral. (default = dermal)')
    parser.add_argument('--inhale_type', type = str, default = 'aerosol', help = 'route of inhale adminstration. aerosol, vapor, gas. (default = aerosol)')
    parser.add_argument()