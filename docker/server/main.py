import sys
import time
import logging

from server import *


def advertise_keys(server, user_num, t, wait_time):
    cnt = 0
    while len(SignatureRequestHandler.U_1) != user_num and cnt < wait_time:
        time.sleep(1)
        cnt += 1

    if len(SignatureRequestHandler.U_1) >= t:
        U_1 = SignatureRequestHandler.U_1
        SecretShareRequestHandler.U_1_num = len(U_1)

        logging.info("{} users have sent signatures".format(len(U_1)))

        server.broadcast_signatures(server.broadcast_port)

        logging.info("online users: " + ','.join(U_1))
    else:
        logging.error("insufficient messages received by the server!")

        sys.exit(1)

    time.sleep(1)

    print("{:=^80s}".format("Finish Advertising Keys"))

    return U_1


def share_keys(server, U_1, t, wait_time):
    cnt = 0
    while len(SecretShareRequestHandler.U_2) != len(U_1) and cnt < wait_time:
        time.sleep(1)
        cnt += 1

    if len(SecretShareRequestHandler.U_2) >= t:
        U_2 = SecretShareRequestHandler.U_2
        MaskingRequestHandler.U_2_num = len(U_2)

        logging.info("{} users have sent ciphertexts".format(len(U_2)))

        for u in U_2:
            msg = pickle.dumps(SecretShareRequestHandler.ciphertexts_map[u])
            server.send(msg, "user" + u, 10001)
    else:
        # the number of the received messages is less than the threshold value for SecretSharing, abort
        logging.error("insufficient ciphertexts received by the server!")

        sys.exit(1)

    time.sleep(1)

    print("{:=^80s}".format("Finish Sharing Keys"))

    return U_2


def masked_input_collection(U_2, t, wait_time):
    cnt = 0
    while len(MaskingRequestHandler.U_3) != len(U_2) and cnt < wait_time:
        time.sleep(1)
        cnt += 1

    if len(MaskingRequestHandler.U_3) >= t:
        U_3 = MaskingRequestHandler.U_3
        ConsistencyRequestHandler.U_3_num = len(U_3)

        logging.info("{} users have sent masked gradients".format(len(U_3)))
    else:
        # the number of the received messages is less than the threshold value for SecretSharing, abort
        logging.error("insufficient masked gradients received by the server!")

        sys.exit(1)

    time.sleep(1)

    print("{:=^80s}".format("Finish Masking Input"))

    return U_3


def consistency_check(server, U_3, t, wait_time):
    msg = pickle.dumps(U_3)
    for u in U_3:
        server.send(msg, "user" + u, 10001)

    time.sleep(0.2)

    cnt = 0
    while len(ConsistencyRequestHandler.U_4) != len(U_3) and cnt < wait_time:
        time.sleep(1)
        cnt += 1

    if len(ConsistencyRequestHandler.U_4) >= t:
        U_4 = ConsistencyRequestHandler.U_4
        UnmaskingRequestHandler.U_4_num = len(U_4)

        logging.info("{} users have sent consistency checks".format(len(U_4)))

        for u in U_4:
            msg = pickle.dumps(ConsistencyRequestHandler.consistency_check_map)
            server.send(msg, "user" + u, 10001)
        
        time.sleep(10)

        if len(ConsistencyRequestHandler.status_list) != 0:
            # at least one user failed in consistency check
            logging.error("consistency check failed: " + ','.join(ConsistencyRequestHandler.status_list))

            sys.exit(2)
        else:
            print("{:=^80s}".format("Finish Consistency Check"))

            return U_4
    else:
        # the number of the received messages is less than the threshold value for SecretSharing, abort
        logging.error("insufficient consistency checks received by the server!")

        sys.exit(1)


def unmasking(server, U_4, shape, t, wait_time):
    cnt = 0
    while len(UnmaskingRequestHandler.U_5) != len(U_4) and cnt < wait_time:
        time.sleep(1)
        cnt += 1

    if len(UnmaskingRequestHandler.U_5) >= t:
        logging.info("{} users have sent shares".format(len(UnmaskingRequestHandler.U_5)))

        output = server.unmask(shape)

        print("{:=^80s}".format("Finish Unmasking"))

        return output

    else:
        # the number of the received messages is less than the threshold value for SecretSharing, abort
        logging.error("insufficient shares received by the server!")

        sys.exit(1)


if __name__ == "__main__":
    user_num = int(sys.argv[1])
    t = int(sys.argv[2])
    wait_time = int(sys.argv[3])

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    server = Server()

    SignatureRequestHandler.user_num = user_num
    server.serve_all()

    U_1 = advertise_keys(server, user_num, t, wait_time)

    U_2 = share_keys(server, U_1, t, wait_time)

    U_3 = masked_input_collection(U_2, t, wait_time)

    # wait for all users to listen to the server
    time.sleep(60)

    U_4 = consistency_check(server, U_3, t, wait_time)

    output = unmasking(server, U_4, (2, 2), t, wait_time)

    print("{:=^80s}".format("Finish Secure Aggregation"))

    print(output)

    server.close_all()

    sys.exit(0)
