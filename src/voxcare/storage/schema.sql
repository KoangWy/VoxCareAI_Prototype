-- VoxCare P1 schema: P1 build guide §4.1 (design D§3.3).
-- TIMESTAMPTZ replaces TIMESTAMP (implementation choice); requested_delivery_window_end resolves G-1.
-- Applied inside the schema named by VOXCARE_DB_SCHEMA (default "voxcare"); see storage/db.py.

CREATE TABLE customer (
  customer_id      VARCHAR(32)  PRIMARY KEY,
  full_name        VARCHAR(100) NOT NULL,
  phone_number     VARCHAR(20)  NOT NULL UNIQUE,
  email            VARCHAR(100),
  default_address  TEXT         NOT NULL,
  created_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE TABLE product (
  product_id          VARCHAR(32)   PRIMARY KEY,            -- e.g. PROD-TV-SAM-01
  product_name        VARCHAR(150)  NOT NULL,
  category            VARCHAR(64)   NOT NULL,               -- ProductCategoryEnum
  brand               VARCHAR(64)   NOT NULL,               -- Samsung|LG|Panasonic|Electrolux|Toshiba
  base_price          DECIMAL(14,2) NOT NULL,
  warranty_months     INT           NOT NULL,
  specifications_json JSONB
);

CREATE TABLE product_variant (
  variant_id        VARCHAR(32)   PRIMARY KEY,              -- e.g. VAR-SAM-55-BLK
  product_id        VARCHAR(32)   NOT NULL REFERENCES product,
  sku               VARCHAR(64)   NOT NULL UNIQUE,
  color             VARCHAR(32),
  storage_capacity  VARCHAR(32),
  screen_size       VARCHAR(32),
  price             DECIMAL(14,2) NOT NULL,
  stock_quantity    INT           NOT NULL                  -- INTERNAL ONLY: never returned by any tool
);

CREATE TABLE orders (
  order_id                 VARCHAR(32)   PRIMARY KEY,       -- 32-char lowercase hex
  order_code               VARCHAR(32)   NOT NULL UNIQUE,   -- ORD-####
  customer_id              VARCHAR(32)   NOT NULL REFERENCES customer,
  status                   VARCHAR(32)   NOT NULL,          -- OrderStatusEnum
  payment_method           VARCHAR(16)   NOT NULL CHECK (payment_method IN ('COD','VNPAY_QR','DOMESTIC_ATM','CREDIT_CARD')),
  payment_status           VARCHAR(16)   NOT NULL CHECK (payment_status IN ('UNPAID','PAID','REFUND_PENDING','REFUNDED')),
  total_amount             DECIMAL(14,2) NOT NULL,
  shipping_address         TEXT          NOT NULL,
  requested_delivery_time  TIMESTAMPTZ,                     -- window START (see G-1)
  requested_delivery_window_end TIMESTAMPTZ,                -- implementation addition, G-1
  estimated_delivery_date  DATE,
  version                  INT           NOT NULL DEFAULT 1,
  created_at               TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
  updated_at               TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE TABLE order_item (
  item_id       VARCHAR(32)   PRIMARY KEY,
  order_id      VARCHAR(32)   NOT NULL REFERENCES orders,
  variant_id    VARCHAR(32)   NOT NULL REFERENCES product_variant,
  product_name  VARCHAR(150)  NOT NULL,
  sku           VARCHAR(64)   NOT NULL,
  quantity      INT           NOT NULL CHECK (quantity > 0),
  unit_price    DECIMAL(14,2) NOT NULL,
  total_price   DECIMAL(14,2) NOT NULL
);

CREATE TABLE shipment (
  shipment_id          VARCHAR(32)  PRIMARY KEY,            -- SHIP-####
  order_id             VARCHAR(32)  NOT NULL UNIQUE REFERENCES orders,
  tracking_code        VARCHAR(64)  NOT NULL UNIQUE,
  carrier_name         VARCHAR(64)  NOT NULL CHECK (carrier_name IN ('GHN','GHTK','ViettelPost')),
  status               VARCHAR(32)  NOT NULL,               -- ShipmentStatusEnum
  driver_name          VARCHAR(100) NOT NULL,
  driver_phone_masked  VARCHAR(20)  NOT NULL,
  origin_hub           VARCHAR(100) NOT NULL,
  destination_address  TEXT         NOT NULL,
  current_location     VARCHAR(150),
  last_update_time     TIMESTAMPTZ  NOT NULL
);

CREATE TABLE shipment_event (
  event_id     VARCHAR(32)  PRIMARY KEY,
  shipment_id  VARCHAR(32)  NOT NULL REFERENCES shipment,
  event_time   TIMESTAMPTZ  NOT NULL,                      -- UTC
  event_type   VARCHAR(64)  NOT NULL,                      -- closed vocabulary, §5
  location     VARCHAR(150) NOT NULL,
  description  TEXT
);

CREATE TABLE installation (
  installation_id  VARCHAR(32)  PRIMARY KEY,
  order_id         VARCHAR(32)  NOT NULL REFERENCES orders, -- non-unique: 1:N
  scheduled_date   DATE         NOT NULL,
  time_slot        VARCHAR(32)  NOT NULL,
  technician_name  VARCHAR(100),
  status           VARCHAR(32)  NOT NULL,                   -- InstallationStatusEnum
  notes            TEXT
);

CREATE TABLE warranty_rma (
  rma_id           VARCHAR(32)  PRIMARY KEY,
  order_item_id    VARCHAR(32)  NOT NULL REFERENCES order_item,
  customer_id      VARCHAR(32)  NOT NULL REFERENCES customer,
  installation_id  VARCHAR(32)  REFERENCES installation,
  diagnosis_class  VARCHAR(64)  NOT NULL,
  symptom_code     VARCHAR(64)  NOT NULL,
  rma_status       VARCHAR(32)  NOT NULL,                   -- RmaStatusEnum
  requested_at     TIMESTAMPTZ  NOT NULL,
  inspection_date  TIMESTAMPTZ
);

CREATE TABLE promotion (
  promotion_id     VARCHAR(32)   PRIMARY KEY,
  voucher_code     VARCHAR(32)   NOT NULL UNIQUE,
  campaign_name    VARCHAR(100)  NOT NULL,
  category         VARCHAR(64)   NOT NULL,
  min_order_value  DECIMAL(14,2) NOT NULL,
  discount_cap     DECIMAL(14,2) NOT NULL,
  valid_from       DATE          NOT NULL,
  valid_to         DATE          NOT NULL,
  is_fictional     BOOLEAN       NOT NULL DEFAULT TRUE
);

CREATE TABLE agent_activity (
  activity_id    VARCHAR(32) PRIMARY KEY,
  session_id     VARCHAR(64) NOT NULL,
  customer_id    VARCHAR(32) NOT NULL REFERENCES customer,
  order_id       VARCHAR(32) REFERENCES orders,
  activity_code  VARCHAR(64) NOT NULL,
  outcome_code   VARCHAR(64) NOT NULL,                      -- 'QUEUED' in Phase 1
  activity_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE audit_log (                                    -- APPEND-ONLY
  audit_id                  UUID        PRIMARY KEY,
  session_id                VARCHAR(64) NOT NULL,
  order_id                  VARCHAR(32) REFERENCES orders,  -- see G-4
  mutation_idempotency_key  UUID,
  actor                     VARCHAR(32) NOT NULL,           -- Caller|Bot|Staff|System
  action_name               VARCHAR(64) NOT NULL,
  action_status             VARCHAR(32) NOT NULL,           -- Success|Failed|Aborted|Denied
  outcome_status            VARCHAR(32) NOT NULL,           -- AuditOutcomeEnum
  request_payload_json      JSONB,                          -- tool_name lives HERE (no tool_name column)
  response_payload_json     JSONB,
  timestamp                 TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Append-only enforcement (implementation choice): the guide's
--   REVOKE UPDATE, DELETE, TRUNCATE ON audit_log FROM voxcare_app;
-- runs inside a block so a database user without CREATEROLE still gets the two rules below.
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'voxcare_app') THEN
    CREATE ROLE voxcare_app NOLOGIN;
  END IF;
  EXECUTE 'REVOKE UPDATE, DELETE, TRUNCATE ON audit_log FROM voxcare_app';
EXCEPTION WHEN insufficient_privilege THEN
  RAISE NOTICE 'voxcare_app role not managed here: %', SQLERRM;
END
$$;
CREATE RULE audit_log_no_update AS ON UPDATE TO audit_log DO INSTEAD NOTHING;
CREATE RULE audit_log_no_delete AS ON DELETE TO audit_log DO INSTEAD NOTHING;

-- Read paths the tools use (implementation choice).
CREATE INDEX orders_customer_idx ON orders (customer_id);
CREATE INDEX order_item_order_idx ON order_item (order_id);
CREATE INDEX shipment_event_shipment_idx ON shipment_event (shipment_id, event_time);
CREATE INDEX audit_log_session_idx ON audit_log (session_id, timestamp);
CREATE INDEX agent_activity_session_idx ON agent_activity (session_id);
