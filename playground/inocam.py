from time import time

import cv2
from amas.agent import OBSERVER, Agent, NotWorkingError
from comprex.agent import ABEND, NEND
from pino.ino import HIGH, LOW, Arduino, PinState

STIMULATOR = "STIMULATOR"
READER = "READER"
RECORDER = "RECORDER"
FILMTAKER = "FILMTAKER"


def timelog():
    anchor = time()

    def inner():
        nonlocal anchor
        now = time()
        print(now - anchor)
        anchor = now

    return inner


async def stimulate(agent: Agent, ino: Arduino):
    LED = 12
    log = timelog()
    try:
        while agent.working():
            for _ in range(5):
                await agent.sleep(2.)
                log()
                ino.digital_write(LED, HIGH)
                agent.send_to(FILMTAKER, HIGH)
                agent.send_to(RECORDER, HIGH)
                await agent.sleep(0.5)
                log()
                ino.digital_write(LED, LOW)
                agent.send_to(FILMTAKER, LOW)
                agent.send_to(RECORDER, LOW)
            break
        agent.finish()
        agent.send_to(OBSERVER, NEND)
    except NotWorkingError:
        agent.send_to(OBSERVER, ABEND)
    return None


async def read(agent: Agent, ino: Arduino):
    try:
        while agent.working():
            v = await agent.call_async(ino.read_until_eol)
            if v is None:
                continue
            s = v.rstrip().decode("utf-8")
            agent.send_to(RECORDER, s)
    except NotWorkingError:
        pass
    # send a message to `RECORDER` to terminate the coroutine
    agent.send_to(RECORDER, NEND)
    return None


async def record(agent: Agent):
    try:
        while agent.working():
            # the following coroutine will hang
            # so it need to receive a message from other agent at the end
            _, mess = await agent.recv()
            print(mess)
    except NotWorkingError:
        pass
    return None


class FilmTaker(Agent):
    def __init__(self, addr: str):
        super().__init__(addr)
        self._led = LOW

    @property
    def led(self) -> PinState:
        return self._led


async def film(agent: FilmTaker, camid: int):
    cap = cv2.VideoCapture(camid)
    try:
        while agent.working():
            await agent.sleep(0.025)

            ret, frame = cap.read()
            if not ret:
                continue

            if agent.led == HIGH:
                cv2.circle(frame, (10, 10), 10, (0, 0, 255), thickness=-1)

            cv2.imshow(f"Camera: {camid}", frame)
            if cv2.waitKey(1) % 0xFF == ord("q"):
                break

        agent.send_to(OBSERVER, NEND)
        agent.finish()
    except NotWorkingError:
        agent.send_to(OBSERVER, ABEND)

    cap.release()
    cv2.destroyAllWindows()
    return None


async def check_pin_state(agent: FilmTaker):
    try:
        while agent.working():
            _, mess = await agent.recv()
            agent._led = mess
    except NotWorkingError:
        pass
    return None


if __name__ == '__main__':
    from amas.connection import Register
    from amas.env import Environment
    from comprex.agent import Observer, _self_terminate
    from pino.ino import OUTPUT, SSINPUT_PULLUP, Comport

    com = Comport() \
        .set_port("/dev/ttyACM0") \
        .connect() \
        .deploy()

    ino = Arduino(com)
    ino.set_pinmode(12, OUTPUT)
    ino.set_pinmode(7, SSINPUT_PULLUP)

    stimulator = Agent(STIMULATOR) \
        .assign_task(stimulate, ino=ino) \
        .assign_task(_self_terminate)

    reader = Agent(READER) \
        .assign_task(read, ino=ino) \
        .assign_task(_self_terminate, ino=ino)

    recorder = Agent(RECORDER) \
        .assign_task(record) \
        .assign_task(_self_terminate)

    filmtaker = FilmTaker(FILMTAKER) \
        .assign_task(film, camid=0) \
        .assign_task(check_pin_state) \
        .assign_task(_self_terminate)

    observer = Observer()

    agents = [stimulator, reader, recorder, observer, filmtaker]
    rgst = Register(agents)
    env_exp = Environment([stimulator, reader, recorder, observer])
    env_cam = Environment([filmtaker])

    try:
        env_cam.parallelize()
        env_exp.run()
        env_cam.join()
    except KeyboardInterrupt:
        observer.send_all(ABEND)
        observer.finish()
