from amas.agent import Agent
from comprex.util import timestamp

RECV = "RECV"
SEND = "SEND"


async def flush_message_for(agent: Agent, duration: float):
    print(timestamp(f"Ignore messages for {duration} sec"))
    while duration >= 0.:
        s, _ = timestamp(None)
        await agent.recv(duration)
        e, _ = timestamp(None)
        duration -= e - s
    print(timestamp("Now, starting to receive messages"))


async def recv_message(agent: Agent):
    await flush_message_for(agent, 5)
    while True:
        addr, mess = await agent.recv()
        if mess == 0:
            break
        print(f"{mess} from {addr}")


async def send_mess(agent: Agent, t: float, n: int):
    for _ in range(n):
        await agent.sleep(t)
        agent.send_to(RECV, 1)
    agent.send_to(RECV, 0)


if __name__ == '__main__':
    from amas.connection import Register
    from amas.env import Environment

    sender = Agent(SEND).assign_task(send_mess, t=1., n=10)
    receiver = Agent(RECV).assign_task(recv_message)

    rgst = Register([sender, receiver])
    env = Environment([sender, receiver])

    env.run()
