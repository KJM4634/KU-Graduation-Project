"""
GPS 모듈 없이도 개발/테스트할 수 있도록, PRD 6장 기준(UART + pyserial/pynmea2, 2초 주기)에
맞춰 가짜 GPS 좌표를 2초마다 생성하는 시뮬레이터.

핵심 설계: 시뮬레이터가 만드는 것은 좌표 자체가 아니라 실제 GPS 모듈이 UART로 보내는 것과
동일한 형식의 NMEA GGA 문장이고, 이를 파싱하는 `parse_nmea_line()`은 실제 하드웨어를 붙일 때도
그대로 재사용한다. 즉 지금 검증하는 파싱 로직이 곧 실기에서 쓸 파싱 로직이다 — 나중에 필요한
건 이 시뮬레이터를 pyserial로 UART에서 읽은 라인으로 바꿔치는 것뿐이다.
"""
import argparse
import random
import time
from dataclasses import dataclass

import pynmea2


@dataclass
class GpsFix:
    lat: float
    lon: float
    timestamp: str


def _to_nmea_lat(lat):
    hemisphere = "N" if lat >= 0 else "S"
    lat = abs(lat)
    deg = int(lat)
    minutes = (lat - deg) * 60
    return f"{deg:02d}{minutes:07.4f}", hemisphere


def _to_nmea_lon(lon):
    hemisphere = "E" if lon >= 0 else "W"
    lon = abs(lon)
    deg = int(lon)
    minutes = (lon - deg) * 60
    return f"{deg:03d}{minutes:07.4f}", hemisphere


def _checksum(body):
    cs = 0
    for ch in body:
        cs ^= ord(ch)
    return f"{cs:02X}"


def generate_fake_nmea_sentence(lat, lon, num_sv=8, hdop=1.0, alt_m=50.0):
    """주어진 위경도로 유효한 $GPGGA 문장(체크섬 포함)을 생성."""
    now = time.gmtime()
    hhmmss = time.strftime("%H%M%S", now) + ".00"
    lat_str, lat_hemi = _to_nmea_lat(lat)
    lon_str, lon_hemi = _to_nmea_lon(lon)
    body = (
        f"GPGGA,{hhmmss},{lat_str},{lat_hemi},{lon_str},{lon_hemi},"
        f"1,{num_sv:02d},{hdop:.1f},{alt_m:.1f},M,0.0,M,,"
    )
    return f"${body}*{_checksum(body)}"


def parse_nmea_line(line: str):
    """실제 GPS 모듈이든 시뮬레이터든 동일하게 통과하는 파싱 함수.
    유효한 fix가 아니면(체크섬 오류, GGA가 아님, gps_qual=0 등) None 반환."""
    try:
        msg = pynmea2.parse(line.strip())
    except pynmea2.ParseError:
        return None
    if not isinstance(msg, pynmea2.types.talker.GGA):
        return None
    if msg.gps_qual is None or int(msg.gps_qual) <= 0:
        return None
    return GpsFix(lat=msg.latitude, lon=msg.longitude, timestamp=str(msg.timestamp))


class FakeGpsSimulator:
    """구조대원이 재난 현장을 천천히 이동하는 상황을 흉내내, 기준 좌표 주변을
    랜덤워크로 조금씩 이동시키며 2초 주기(PRD 기준)로 GPS fix를 만든다."""

    def __init__(self, start_lat=37.5665, start_lon=126.9780, interval_sec=2.0, step_deg=0.00005):
        self.lat = start_lat
        self.lon = start_lon
        self.interval_sec = interval_sec
        self.step_deg = step_deg

    def next_fix(self) -> GpsFix:
        self.lat += random.uniform(-1, 1) * self.step_deg
        self.lon += random.uniform(-1, 1) * self.step_deg
        sentence = generate_fake_nmea_sentence(self.lat, self.lon)
        fix = parse_nmea_line(sentence)
        assert fix is not None, f"시뮬레이터가 생성한 문장을 파싱하지 못함: {sentence}"
        return fix

    def stream(self, count=None):
        """`interval_sec`마다 GpsFix를 yield하는 제너레이터. count=None이면 무한 반복."""
        i = 0
        while count is None or i < count:
            yield self.next_fix()
            i += 1
            if count is None or i < count:
                time.sleep(self.interval_sec)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-lat", type=float, default=37.5665)
    parser.add_argument("--start-lon", type=float, default=126.9780)
    parser.add_argument("--interval", type=float, default=2.0, help="fix 생성 주기(초), PRD 기본값 2.0")
    parser.add_argument("--count", type=int, default=5, help="생성할 fix 개수 (0=무한)")
    args = parser.parse_args()

    sim = FakeGpsSimulator(args.start_lat, args.start_lon, interval_sec=args.interval)
    count = None if args.count == 0 else args.count
    print(f"[GPS 시뮬레이터] {args.interval}초 주기로 fix 생성 시작 (count={count or '무한'})")
    for i, fix in enumerate(sim.stream(count=count)):
        print(f"  fix {i}: lat={fix.lat:.6f}, lon={fix.lon:.6f}, time={fix.timestamp}")


if __name__ == "__main__":
    main()
