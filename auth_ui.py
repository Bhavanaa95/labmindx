import streamlit as st
from auth import (
    authenticate_user,
    create_user,
    validate_signup,
)


def render_auth_page() -> None:
    """Render the LabMind login and signup interface."""

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            display: none;
        }

        .auth-brand-panel {
            padding: 38px;
            border-radius: 30px;
            background: linear-gradient(
                135deg,
                rgba(37, 99, 235, 0.98),
                rgba(124, 58, 237, 0.98)
            );
            box-shadow: 0 30px 80px rgba(15, 23, 42, 0.30);
        }

        .auth-brand {
            color: white;
            font-size: 46px;
            line-height: 1.1;
            font-weight: 1000;
            margin-bottom: 8px;
        }

        .auth-tagline {
            color: #DBEAFE;
            font-size: 18px;
            font-weight: 800;
            margin-bottom: 32px;
        }

        .auth-heading {
            color: white;
            font-size: 31px;
            line-height: 1.25;
            font-weight: 1000;
            margin-bottom: 12px;
        }

        .auth-description {
            color: #E0E7FF;
            font-size: 16px;
            line-height: 1.7;
            margin-bottom: 24px;
        }

        .auth-benefit {
            padding: 13px 15px;
            margin-bottom: 10px;
            border-radius: 14px;
            color: white;
            font-weight: 750;
            background: rgba(15, 23, 42, 0.18);
            border: 1px solid rgba(255, 255, 255, 0.16);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    left_column, right_column = st.columns(
        [1.05, 0.95],
        gap="large",
    )

    with left_column:
     st.markdown(
        '<div style="padding:38px;border-radius:30px;'
        'background:linear-gradient(135deg,#2563eb,#7c3aed);'
        'box-shadow:0 30px 80px rgba(15,23,42,.30);color:white;">'
        '<div style="font-size:46px;font-weight:900;margin-bottom:8px;">'
        '🧪 LabMind.ai</div>'
        '<div style="font-size:18px;font-weight:700;'
        'color:#dbeafe;margin-bottom:32px;">Premium AI Data Analyst</div>'
        '<div style="font-size:31px;font-weight:900;'
        'margin-bottom:12px;">Turn datasets into decisions.</div>'
        '<div style="font-size:16px;line-height:1.7;'
        'color:#e0e7ff;margin-bottom:24px;">'
        'Upload data, inspect quality, clean missing values, '
        'compare machine-learning models, explain predictions, '
        'and generate professional reports.</div>'
        '<div style="padding:13px 15px;margin-bottom:10px;'
        'border-radius:14px;background:rgba(15,23,42,.18);'
        'border:1px solid rgba(255,255,255,.16);font-weight:700;">'
        '📊 Automated dataset intelligence</div>'
        '<div style="padding:13px 15px;margin-bottom:10px;'
        'border-radius:14px;background:rgba(15,23,42,.18);'
        'border:1px solid rgba(255,255,255,.16);font-weight:700;">'
        '🤖 Multi-model AutoML comparison</div>'
        '<div style="padding:13px 15px;margin-bottom:10px;'
        'border-radius:14px;background:rgba(15,23,42,.18);'
        'border:1px solid rgba(255,255,255,.16);font-weight:700;">'
        '🧠 Explainable AI insights</div>'
        '<div style="padding:13px 15px;'
        'border-radius:14px;background:rgba(15,23,42,.18);'
        'border:1px solid rgba(255,255,255,.16);font-weight:700;">'
        '📄 Executive-ready reports</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    with right_column:
        login_tab, signup_tab = st.tabs(
            ["🔐 Login", "✨ Create Account"]
        )

        with login_tab:
            st.markdown("## Welcome back")
            st.caption(
                "Sign in to continue to your LabMind workspace."
            )

            with st.form("login_form"):
                login_email = st.text_input(
                    "Email address",
                    placeholder="you@example.com",
                )

                login_password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Enter your password",
                )

                login_submitted = st.form_submit_button(
                    "Log in to LabMind",
                    use_container_width=True,
                )

            if login_submitted:
                success, user, message = authenticate_user(
                    login_email,
                    login_password,
                )

                if success:
                    st.session_state["authenticated"] = True
                    st.session_state["current_user"] = user
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

        with signup_tab:
            st.markdown("## Create your account")
            st.caption(
                "Set up your LabMind workspace in under a minute."
            )

            with st.form("signup_form"):
                signup_name = st.text_input(
                    "Full name",
                    placeholder="Your full name",
                )

                signup_email = st.text_input(
                    "Email address",
                    placeholder="you@example.com",
                    key="signup_email",
                )

                signup_password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Minimum 8 characters",
                    key="signup_password",
                )

                signup_confirm_password = st.text_input(
                    "Confirm password",
                    type="password",
                    placeholder="Re-enter your password",
                )

                signup_submitted = st.form_submit_button(
                    "Create LabMind account",
                    use_container_width=True,
                )

            if signup_submitted:
                valid, validation_message = validate_signup(
                    signup_name,
                    signup_email,
                    signup_password,
                    signup_confirm_password,
                )

                if not valid:
                    st.error(validation_message)
                else:
                    success, message = create_user(
                        signup_name,
                        signup_email,
                        signup_password,
                    )

                    if success:
                        st.success(
                            "Account created successfully. "
                            "Open the Login tab and sign in."
                        )
                    else:
                        st.error(message)