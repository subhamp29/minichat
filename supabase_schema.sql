-- Supabase schema for MiniChat / Bhavyam AI
-- Run this in the Supabase SQL Editor after creating a new project.
-- Requires: Auth enabled in Supabase (Authentication → Providers).

-- 1. Conversations table (per-user)
create table if not exists public.conversations (
  id text primary key,
  user_id uuid not null references auth.users(id) on delete cascade,
  title text not null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

-- 2. Messages table (immutable append-only log, per-user via conversation)
create table if not exists public.messages (
  id bigserial primary key,
  conversation_id text not null references public.conversations(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  role text not null check (role in ('user','assistant','system')),
  content text not null,
  model text not null,
  backend text not null check (backend in ('remote','local_gguf','n8n','n8n_orchestrated')),
  tokens_in integer,
  tokens_out integer,
  created_at timestamptz not null default now()
);

-- 3. Indexes for fast history lookups
create index if not exists idx_messages_conversation
  on public.messages (conversation_id, created_at);

create index if not exists idx_conversations_user
  on public.conversations (user_id, updated_at desc);

create index if not exists idx_messages_user
  on public.messages (user_id, created_at);

-- 4. Auto-update updated_at on conversation changes
create or replace function public.set_updated_at()
returns trigger as $$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;

drop trigger if exists trg_set_updated_at on public.conversations;
create trigger trg_set_updated_at
  before update on public.conversations
  for each row execute function public.set_updated_at();

-- 5. Row Level Security (RLS)
alter table public.conversations enable row level security;
alter table public.messages enable row level security;

-- Users can view their own conversations
create policy "Users can view own conversations"
  on public.conversations for select
  using (auth.uid() = user_id);

-- Users can insert their own conversations
create policy "Users can insert own conversations"
  on public.conversations for insert
  with check (auth.uid() = user_id);

-- Users can update their own conversations
create policy "Users can update own conversations"
  on public.conversations for update
  using (auth.uid() = user_id);

-- Users can delete their own conversations
create policy "Users can delete own conversations"
  on public.conversations for delete
  using (auth.uid() = user_id);

-- Users can view their own messages
create policy "Users can view own messages"
  on public.messages for select
  using (auth.uid() = user_id);

-- Users can insert their own messages
create policy "Users can insert own messages"
  on public.messages for insert
  with check (auth.uid() = user_id);

-- 6. Realtime publication (optional, for live UI updates)
alter publication supabase_realtime add table public.messages;
alter publication supabase_realtime add table public.conversations;
