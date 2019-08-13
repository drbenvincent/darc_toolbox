def was_last_response_B(last_keypress):
    '''this function returns last_response_chose_delayed as True or False,
    taking into account the location of the immediate and delayed rewards.
    last_keypress is either 'left' or 'right'
    '''
    if last_keypress == 'left':
        last_response_chose_B = False
    elif last_keypress == 'right':
        last_response_chose_B = True
    else:
        raise Exception('unexpected last_keypress')
    return last_response_chose_B


def days_to_string(delay_days):
    '''convert a number of days to a sensible string'''
    DAYS_PER_MONTH = 30

    if delay_days == 0:
        delay_text = f'now'

    elif delay_days < 1/24:
        mins = delay_days*(24*60)
        if mins == 1:
            delay_text = f'in {mins:0.0f} minuite'
        else:
            if mins.is_integer():
                delay_text = f'in {mins:0.0f} minuites'
            else:
                delay_text = f'in {mins:0.2f} minuites'

    elif delay_days < 1:
        hours = delay_days*24
        if hours == 1:
            delay_text = f'in {hours:0.0f} hour'
        else:
            if hours.is_integer():
                delay_text = f'in {hours:0.0f} hours'
            else:
                delay_text = f'in {hours:0.2f} hours'

    elif delay_days < 7:
        days = delay_days
        if days == 1:
            delay_text = f'in {days:0.0f} day'
        else:
            if days.is_integer():
                delay_text = f'in {days:0.0f} days'
            else:
                delay_text = f'in {days:0.2f} days'

    elif delay_days < 30:
        weeks = delay_days/7
        if weeks == 1:
            delay_text = f'in {weeks:0.0f} week'
        else:
            if weeks.is_integer():
                delay_text = f'in {weeks:0.0f} weeks'
            else:
                delay_text = f'in {weeks:0.2f} weeks'

    elif delay_days < 365:
        months = delay_days/DAYS_PER_MONTH
        if months == 1:
            delay_text = f'in {months:0.0f} month'
        else:
            if months.is_integer():
                delay_text = f'in {months:0.0f} months'
            else:
                delay_text = f'in {months:0.2f} months'

    else:
        years = delay_days/365
        if years == 1:
            delay_text = f'in {years:0.0f} year'
        else:
            if years.is_integer():
                delay_text = f'in {years:0.0f} years'
            else:
                delay_text = f'in {years:0.2f} years'

    return delay_text
