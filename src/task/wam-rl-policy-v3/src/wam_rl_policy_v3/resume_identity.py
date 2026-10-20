"""Strict PPO resumes with one explicit physical-backend migration boundary."""


def validate_resume_identity(saved, current, *, migrate_physics=False):
    if saved == current:
        return None
    missing = object()
    changed = {key for key in saved.keys() | current.keys()
               if saved.get(key, missing) != current.get(key, missing)}
    backends = {'cpu', 'gpu', 'gpu-cpu'}
    if (migrate_physics and changed == {'physics_backend'}
            and saved.get('physics_backend') in backends
            and current.get('physics_backend') in backends):
        return dict(kind='explicit_physics_backend',
                    source=saved['physics_backend'], target=current['physics_backend'])
    raise ValueError('PPO resume identity changed: ' + ', '.join(sorted(changed)))
