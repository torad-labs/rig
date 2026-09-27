# local/

Everything rig fetched or built on this machine, and nothing else. It exists on every machine
rig runs on: this checkout, a user's install, a rented box (`<remote_dir>/local/`). Gitignored
except this file; never `/tmp`, never a cache dir. `src/shared/layout.ts` is the one place that
names these paths, here and on a box.

```
packs/<head>/            the source pack, the served pack, the draft head — each trusted only
                         after its sha256 matched (rig fetch, rig derive)
engine-builds/<sha7>-sm<cap>/   a published build of the engine for one commit and one card:
                         complete or absent, the BUILD marker written last (rig build); the
                         unit's ExecStart points here
engine-build-trees/      the cmake trees those builds came from; a cache, rebuildable
engine-sources/          the fork at the pin when the submodule is not (a box has no repo)
logs/                    the head's log, build and configure logs
gate-runs/<head>/<run>/  a gate run's results and summary.json (rig gate); HumanEval.jsonl
rented-box/              the rented box's state, cached build tarballs, pulled runs (rig vast)
calibration/             the calibration recipe's corpus and outputs (scripts/engine-corpus.ts)
server.pid               the pid of a server started outside systemd (a box)
```

**Belongs here:** what a rig command wrote, at the path `layout.ts` gave it. A running
llama-server has its pack mmapped: nothing writes a served path in place, ever — a sibling
file, then a rename.

**Does not belong here:** anything hand-made (an experiment, a script, a copy of a result: the
project wrapper's `research/`, or `heads/<head>/evidence/` once cited), anything git should
hold, a symlink into `/tmp` or `~/.cache`.

Deleting from here is safe when nothing runs out of it: `systemctl --user status
rig-<head>.service` names the build the head runs on.
