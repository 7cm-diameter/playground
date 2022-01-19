import cv2
from amas.agent import Agent, NotWorkingError
from comprex.agent import ABEND, NEND, OBSERVER, RECORDER
from pino.ino import HIGH, LOW


class Recorder(Agent):
    def __init__(self, camid: int):
        super().__init__(RECORDER)
        self.camid = camid
        self.pin_state = LOW


async def record(agent: Recorder):
    cap = cv2.VideoCapture(agent.camid)

    try:
        while agent.working():
            await agent.sleep(0.025)

            ret, frame = cap.read()
            if not ret:
                continue

            if agent.pin_state == HIGH:
                cv2.circle(frame, (10, 10), 10, (0, 0, 255), thickness=-1)

            cv2.imshow(f"Camera: {agent.camid}", frame)
            if cv2.waitKey(1) % 0xFF == ord("q"):
                break

        agent.send_to(OBSERVER, NEND)
        agent.finish()
    except NotWorkingError:
        agent.send_to(OBSERVER, ABEND)
        agent.finish()

    cap.release()
    cv2.destroyAllWindows()


async def update_pin_state(agent: Recorder):
    try:
        while agent.working():
            _, mess = await agent.recv()
            agent.pin_state = mess
    except NotWorkingError:
        pass
    return None


async def read(agent: Agent):
    try:
        for _ in range(10):
            await agent.sleep(1)
            agent.send_to(RECORDER, HIGH)
            await agent.sleep(1)
            agent.send_to(RECORDER, LOW)
    except NotWorkingError:
        pass


if __name__ == '__main__':
    from amas.connection import Register
    from amas.env import Environment
    from comprex.agent import Observer, _self_terminate

    recorder = Recorder(0) \
        .assign_task(record) \
        .assign_task(update_pin_state) \
        .assign_task(_self_terminate)
    reader = Agent("READER") \
        .assign_task(read) \
        .assign_task(_self_terminate)
    observer = Observer()

    register = Register([recorder, reader, observer])
    env = Environment([recorder, reader, observer])
    env.run()
