"""empty message

Revision ID: 6a8f04ef5e45
Revises: 
Create Date: 2023-12-30 05:59:27.194336

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '6a8f04ef5e45'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('user',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('username', sa.String(length=80), nullable=False),
    sa.Column('password', sa.String(length=80), nullable=False),
    sa.Column('created_date', sa.DateTime(), nullable=True),
    sa.Column('country', sa.String(length=2), nullable=False),
    sa.Column('birth_day', sa.Date(), nullable=False),
    sa.Column('is_locked', sa.Boolean(), nullable=True),
    sa.Column('money_in_no', sa.Float(), nullable=True),
    sa.Column('money_out_no', sa.Float(), nullable=True),
    sa.Column('money_in_gbp', sa.Float(), nullable=True),
    sa.Column('money_out_gbp', sa.Float(), nullable=True),
    sa.Column('money_in_day_no', sa.Float(), nullable=True),
    sa.Column('money_out_day_no', sa.Float(), nullable=True),
    sa.Column('money_in_sum', sa.Float(), nullable=True),
    sa.Column('money_out_sum', sa.Float(), nullable=True),
    sa.Column('money_in_day_sum', sa.Float(), nullable=True),
    sa.Column('money_out_day_sum', sa.Float(), nullable=True),
    sa.Column('balance_no', sa.Float(), nullable=True),
    sa.Column('balance_day_no', sa.Float(), nullable=True),
    sa.Column('balance_sum', sa.Float(), nullable=True),
    sa.Column('balance_day_sum', sa.Float(), nullable=True),
    sa.Column('balance', sa.Float(), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('username')
    )
    op.create_table('transaction',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('type', sa.String(length=20), nullable=False),
    sa.Column('state', sa.String(length=20), nullable=False),
    sa.Column('created_date', sa.DateTime(), nullable=True),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('amount_gb', sa.Float(), nullable=False),
    sa.Column('currency', sa.String(length=10), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('transaction')
    op.drop_table('user')
    # ### end Alembic commands ###
