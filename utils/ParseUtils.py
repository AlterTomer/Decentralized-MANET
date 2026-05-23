def parse_tx_rx_data(raw_tx: str, raw_rx: str) -> tuple:
    """
    Parse Tx, Rx nodes data from settings.

    Example raw data: '1 ,3 ,9'/ '2'
    Parameters
    -------
    raw_tx: Tx nodes from config file, at least one node.
    raw_rx: Rx nodes from config file, at least one node.

    Return
    -------
    Tuple of Int/List representing the transmitting and receiving nodes.
    """
    if raw_tx.startswith("[") and raw_tx.endswith("]"):
        tx = [int(x) for x in raw_tx[1:-1].replace(" ", "").split(",") if x]
    elif "," in raw_tx:
        tx = [int(x) for x in raw_tx.replace(" ", "").split(",") if x]
    else:
        tx = int(raw_tx)

    if raw_rx.startswith("[") and raw_rx.endswith("]"):
        rx = [int(x) for x in raw_rx[1:-1].replace(" ", "").split(",") if x]
    elif "," in raw_rx:
        rx = [int(x) for x in raw_rx.replace(" ", "").split(",") if x]
    else:
        rx = int(raw_rx)

    return tx, rx
