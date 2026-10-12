"""Freeze balanced held-session coverage before fitting, using source clocks."""


def balanced_video_panel(records, per_session=4):
    groups = {}
    for index, record in enumerate(records):
        if record['split'] != 'val':
            raise ValueError('video validation panel contains training windows')
        session = record['session']
        session = session[:-4] if session.endswith('.mov') else session
        groups.setdefault(session, []).append(index)
    if not groups or len(groups) > 4 or per_session < 1:
        raise ValueError('one to four held sessions and positive clip budget required')
    panel = []
    for session in sorted(groups):
        indices = sorted(groups[session], key=lambda i:(records[i]['source_current_frame'],records[i]['path']))
        count = min(per_session,len(indices))
        positions = [0] if count == 1 else [j*(len(indices)-1)//(count-1) for j in range(count)]
        for position in positions:
            i = indices[position]
            panel.append(dict(index=i,session=session,path=records[i]['path'],sha256=records[i]['sha256'],
                              source_current_frame=records[i]['source_current_frame']))
    return panel
