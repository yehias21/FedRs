# Secure Aggregation

This is an unofficial implementation of Secure Aggregation Protocol. The details of the protocol can be found in the original paper: [(CCS '17) Practical Secure Aggregation for Privacy-Preserving Machine Learning](https://dl.acm.org/doi/abs/10.1145/3133956.3133982).

Based on Secure Aggregation Protocol, a verifiable secure aggregation protocol for cross-device federated learning is proposed. The details of the protocol can be found in the original paper: [(TDSC 2021) VERSA: Verifiable Secure Aggregation for Cross-Device Federated Learning](https://ieeexplore.ieee.org/abstract/document/9609695).

## Usage

There are two ways to use the Secure Aggregation Protocol.

### Docker
---

This is the recommended option, as all entities are independent containers. That is, a real federated learning scenario is simulated in this way.

- Pull base image:
```
$ docker pull chenjunbao/secureaggregation
```

- Build docker images for each entity:
```
$ git clone https://github.com/chen-junbao/secureaggregation.git
$ cd secureaggregation/docker
$ ./scripts/build.sh
```

- Simulate 100 users and set the waiting time to 60 seconds:
```
$ ./start.sh -u 100 -t 60
```

### Single Machine
---

- Install python libraries:

```
$ git clone https://github.com/chen-junbao/secureaggregation.git
$ cd secureaggregation
$ pip install -r requirements.txt
$ pip install git+https://github.com/blockstack/secret-sharing

$ python main.py -h
```

- Simulate 100 users and set the waiting time to 300 seconds:
```

VERSA:
```
git checkout versa
python main.py -u 100 -t 300
```
