"""
2D Vizualizace controls_challenge
Spuštění: python visualize.py --data_path ./data/00000.csv --controller pid
"""

import argparse
import sys
import os
import numpy as np
import pygame

# ── Zajistí že Python najde tinyphysics v kořeni projektu ──────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import tinyphysics
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator, CONTROL_START_IDX
import importlib

# ── Konstanty ──────────────────────────────────────────────────────────────────
FPS_GAME      = 60          # fps pygame
SIM_STEPS_PER_FRAME = 1     # kolik kroků simulace na jeden frame
MODEL_PATH    = "./models/tinyphysics.onnx"

# ── Barvy (dark terminal theme) ────────────────────────────────────────────────
BG          = (10,  12,  18)
GRID        = (25,  30,  45)
ROAD_BG     = (20,  24,  36)
ROAD_EDGE   = (40,  50,  75)
TARGET_COL  = (0,  200, 120)   # zelená – cílová trať
CURRENT_COL = (255, 80,  80)   # červená – skutečná poloha
FUTURE_COL  = (80, 140, 255)   # modrá – future_plan preview
CAR_COL     = (255, 220,  50)  # žlutá – auto
HUD_BG      = (15,  18,  28, 200)
WHITE       = (230, 235, 245)
GRAY        = (100, 115, 140)
GREEN       = (0,  220, 130)
RED         = (255,  70,  70)
CYAN        = (0,  200, 220)

# ── Rozložení obrazovky ────────────────────────────────────────────────────────
W, H        = 1280, 720
TRACK_X     = 220            # levý okraj trati
TRACK_W     = 860            # šířka trati
TRACK_H     = H - 40         # výška trati
ROAD_HALF   = 90             # px půlšířka silnice
FUTURE_LEN  = 50             # kolik budoucích kroků zobrazit


def load_controller(name: str):
    mod = importlib.import_module(f"controllers.{name}")
    return mod.Controller()


def lat_to_px(lat: float, scale: float = 60.0) -> int:
    """Laterální zrychlení [m/s²] → pixel offset od středu silnice."""
    return int(np.clip(lat * scale, -ROAD_HALF + 10, ROAD_HALF - 10))


def step_to_x(step: int, total: int) -> int:
    """Krok simulace → x souřadnice na obrazovce (zleva doprava)."""
    return TRACK_X + int(step / max(total - 1, 1) * TRACK_W)


def draw_grid(surf):
    for x in range(TRACK_X, TRACK_X + TRACK_W, 60):
        pygame.draw.line(surf, GRID, (x, 0), (x, H), 1)
    for y in range(0, H, 60):
        pygame.draw.line(surf, GRID, (0, y), (W, y), 1)


def draw_road(surf, mid_y):
    road_rect = pygame.Rect(TRACK_X, mid_y - ROAD_HALF, TRACK_W, ROAD_HALF * 2)
    pygame.draw.rect(surf, ROAD_BG, road_rect)
    pygame.draw.line(surf, ROAD_EDGE, (TRACK_X, mid_y - ROAD_HALF),
                     (TRACK_X + TRACK_W, mid_y - ROAD_HALF), 2)
    pygame.draw.line(surf, ROAD_EDGE, (TRACK_X, mid_y + ROAD_HALF),
                     (TRACK_X + TRACK_W, mid_y + ROAD_HALF), 2)
    # Středová přerušovaná čára
    for x in range(TRACK_X, TRACK_X + TRACK_W, 30):
        pygame.draw.line(surf, GRID, (x, mid_y), (x + 15, mid_y), 1)


def draw_polyline(surf, points, color, width=2, alpha=255):
    if len(points) < 2:
        return
    if alpha < 255:
        temp = pygame.Surface((W, H), pygame.SRCALPHA)
        pygame.draw.lines(temp, (*color, alpha), False, points, width)
        surf.blit(temp, (0, 0))
    else:
        pygame.draw.lines(surf, color, False, points, width)


def draw_car(surf, x, y):
    CW, CH = 18, 30
    car_surf = pygame.Surface((CW, CH), pygame.SRCALPHA)
    # Karoserie
    pygame.draw.rect(car_surf, CAR_COL, (2, 4, CW-4, CH-8), border_radius=4)
    # Okna
    pygame.draw.rect(car_surf, (30, 40, 60), (4, 7, CW-8, 8), border_radius=2)
    # Světla
    pygame.draw.rect(car_surf, (255, 255, 180), (2, CH-10, 5, 3))
    pygame.draw.rect(car_surf, (255, 255, 180), (CW-7, CH-10, 5, 3))
    surf.blit(car_surf, (x - CW//2, y - CH//2))


def draw_hud(surf, font_big, font_med, font_sm, info: dict):
    # Levý panel
    panel = pygame.Surface((200, H), pygame.SRCALPHA)
    panel.fill((12, 15, 25, 220))
    surf.blit(panel, (0, 0))

    pygame.draw.line(surf, ROAD_EDGE, (200, 0), (200, H), 1)

    y = 20
    title = font_big.render("CONTROLS", True, CYAN)
    surf.blit(title, (10, y)); y += 30
    title2 = font_big.render("CHALLENGE", True, CYAN)
    surf.blit(title2, (10, y)); y += 40

    def row(label, val, color=WHITE):
        nonlocal y
        lbl = font_sm.render(label, True, GRAY)
        surf.blit(lbl, (10, y))
        vl = font_med.render(str(val), True, color)
        surf.blit(vl, (10, y + 14))
        y += 38

    row("CONTROLLER", info.get("controller", "—"), CYAN)
    row("STEP", f"{info.get('step', 0)} / {info.get('total', 0)}")

    progress = info.get('step', 0) / max(info.get('total', 1), 1)
    bar_w = 180
    pygame.draw.rect(surf, GRID, (10, y, bar_w, 8), border_radius=4)
    pygame.draw.rect(surf, CYAN, (10, y, int(bar_w * progress), 8), border_radius=4)
    y += 20

    lat_cost  = info.get("lat_cost", 0.0)
    jerk_cost = info.get("jerk_cost", 0.0)
    total     = info.get("total_cost", 0.0)

    col_lat  = GREEN if lat_cost  < 0.5 else (RED if lat_cost  > 1.5 else WHITE)
    col_jerk = GREEN if jerk_cost < 0.5 else (RED if jerk_cost > 1.5 else WHITE)
    col_tot  = GREEN if total     < 1.0 else (RED if total     > 3.0 else WHITE)

    row("LAT COST",   f"{lat_cost:.4f}",  col_lat)
    row("JERK COST",  f"{jerk_cost:.4f}", col_jerk)
    row("TOTAL COST", f"{total:.4f}",     col_tot)
    row("vEgo (m/s)", f"{info.get('vego', 0):.1f}")
    row("target lat", f"{info.get('target', 0):.3f}")
    row("current lat",f"{info.get('current', 0):.3f}")
    row("error",      f"{info.get('error', 0):.3f}",
        GREEN if abs(info.get('error', 0)) < 0.1 else RED)

    # Legenda
    y = H - 120
    for color, label in [(TARGET_COL, "Target"), (CURRENT_COL, "Current"),
                         (FUTURE_COL, "Future plan"), (CAR_COL, "Car")]:
        pygame.draw.rect(surf, color, (10, y + 4, 14, 8), border_radius=2)
        lbl = font_sm.render(label, True, WHITE)
        surf.blit(lbl, (30, y))
        y += 22

    # Klávesy
    hint = font_sm.render("R=restart  Q=quit  SPACE=pause", True, GRAY)
    surf.blit(hint, (W - hint.get_width() - 10, H - 20))


def run(data_path: str, controller_name: str, speed: int = 1):
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption(f"controls_challenge viz – {controller_name}")
    clock = pygame.time.Clock()

    font_big = pygame.font.SysFont("monospace", 16, bold=True)
    font_med = pygame.font.SysFont("monospace", 14, bold=True)
    font_sm  = pygame.font.SysFont("monospace", 11)

    def init_sim():
        ctrl = load_controller(controller_name)
        model = TinyPhysicsModel(MODEL_PATH, debug=False)
        sim = TinyPhysicsSimulator(model, data_path, controller=ctrl, debug=False)
        # Přeskočíme warm-up kroky
        for _ in range(CONTROL_START_IDX):
            sim.step()
        return sim, ctrl

    sim, ctrl = init_sim()
    total_steps = len(sim.data)

    # Detekuj názvy sloupců automaticky
    cols = sim.data.columns.tolist()
    print(f"[INFO] CSV sloupce: {cols}")
    vcol      = next((c for c in cols if c.lower() in ('vego', 'v_ego', 'velocity')), None)
    tlatcol   = next((c for c in cols if 'target' in c.lower() and 'lat' in c.lower()), None)
    if not tlatcol:
        tlatcol = next((c for c in cols if 'lataccel' in c.lower()), cols[0])
    print(f"[INFO] vEgo={vcol}  target_lataccel={tlatcol}")

    mid_y = H // 2   # střed silnice na obrazovce

    # Historie pro kreslení čar
    target_hist:  list[tuple] = []
    current_hist: list[tuple] = []
    future_pts:   list[tuple] = []

    step       = CONTROL_START_IDX
    paused     = False
    done       = False
    lat_cost   = 0.0
    jerk_cost  = 0.0
    total_cost = 0.0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_r:
                    sim, ctrl = init_sim()
                    step = CONTROL_START_IDX
                    target_hist.clear(); current_hist.clear(); future_pts.clear()
                    done = False; lat_cost = jerk_cost = total_cost = 0.0
                if event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    speed = min(speed + 1, 10)
                if event.key == pygame.K_MINUS:
                    speed = max(speed - 1, 1)

        # ── Simulační krok ─────────────────────────────────────────────────────
        if not paused and not done:
            for _ in range(speed):
                if step >= total_steps - 1:
                    done = True
                    break
                try:
                    sim.step()
                    step += 1
                except Exception:
                    done = True
                    break

            # Přečti aktuální stav ze simulátoru
            idx = min(step, len(sim.target_lataccel_history) - 1)
            if idx >= 0 and len(sim.target_lataccel_history) > 0:
                t_lat = sim.target_lataccel_history[idx]
                c_lat = sim.current_lataccel_history[idx]

                sx = step_to_x(step, total_steps)
                target_hist.append((sx, mid_y - lat_to_px(t_lat)))
                current_hist.append((sx, mid_y - lat_to_px(c_lat)))

                # Future plan z dat (target_lataccel sloupec)
                future_pts.clear()
                future_end = min(step + FUTURE_LEN, total_steps)
                for fs in range(step, future_end):
                    fx = step_to_x(fs, total_steps)
                    f_lat = sim.data[tlatcol].iloc[fs]
                    future_pts.append((fx, mid_y - lat_to_px(f_lat)))

                # Cost výpočet
                if len(sim.target_lataccel_history) > 1:
                    t_arr = np.array(sim.target_lataccel_history)
                    c_arr = np.array(sim.current_lataccel_history)
                    lat_cost   = float(np.mean((t_arr - c_arr)**2) * 100)
                    jerks      = np.diff(c_arr) * tinyphysics.FPS
                    jerk_cost  = float(np.mean(jerks**2) * 100) if len(jerks) > 0 else 0.0
                    total_cost = lat_cost * 50 + jerk_cost

        # ── Kreslení ───────────────────────────────────────────────────────────
        screen.fill(BG)
        draw_grid(screen)
        draw_road(screen, mid_y)

        # Future plan (tenká modrá přerušovaná)
        if len(future_pts) > 1:
            draw_polyline(screen, future_pts, FUTURE_COL, 1, 120)

        # Target trasa (zelená)
        if len(target_hist) > 1:
            draw_polyline(screen, target_hist, TARGET_COL, 2)

        # Skutečná trasa (červená)
        if len(current_hist) > 1:
            draw_polyline(screen, current_hist, CURRENT_COL, 2)

        # Auto na aktuální pozici
        if len(current_hist) > 0:
            cx, cy = current_hist[-1]
            draw_car(screen, cx, cy)

        # Vertikální čára – aktuální krok
        sx = step_to_x(step, total_steps)
        pygame.draw.line(screen, (60, 70, 100), (sx, 0), (sx, H), 1)

        # DONE banner
        if done:
            banner = font_big.render(
                f"DONE  total_cost={total_cost:.4f}  (R = restart)", True, CYAN)
            bx = TRACK_X + TRACK_W // 2 - banner.get_width() // 2
            screen.blit(banner, (bx, H // 2 - 12))

        # Speed indicator
        sp_txt = font_sm.render(f"speed ×{speed}  (+/-)", True, GRAY)
        screen.blit(sp_txt, (TRACK_X + 10, 10))

        # Aktuální hodnoty pro HUD – detekuj název sloupce automaticky
        vcol = next((c for c in sim.data.columns if c.lower() in ('vego', 'v_ego', 'velocity')), None)
        vego = float(sim.data[vcol].iloc[min(step, total_steps-1)]) if vcol else 0.0
        t_now   = sim.target_lataccel_history[-1] if sim.target_lataccel_history else 0
        c_now   = sim.current_lataccel_history[-1] if sim.current_lataccel_history else 0

        draw_hud(screen, font_big, font_med, font_sm, {
            "controller": controller_name,
            "step":       step,
            "total":      total_steps,
            "lat_cost":   lat_cost,
            "jerk_cost":  jerk_cost,
            "total_cost": total_cost,
            "vego":       vego,
            "target":     t_now,
            "current":    c_now,
            "error":      t_now - c_now,
        })

        pygame.display.flip()
        clock.tick(FPS_GAME)

    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D vizualizace controls_challenge")
    parser.add_argument("--data_path",   required=True,  help="Cesta k CSV souboru, např. ./data/00000.csv")
    parser.add_argument("--controller",  default="pid",  help="Název kontroleru (složka controllers/)")
    parser.add_argument("--speed",       type=int, default=1, help="Počáteční rychlost simulace (1–10)")
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"[ERROR] Soubor nenalezen: {args.data_path}")
        sys.exit(1)
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model nenalezen: {MODEL_PATH}")
        sys.exit(1)

    run(args.data_path, args.controller, args.speed)