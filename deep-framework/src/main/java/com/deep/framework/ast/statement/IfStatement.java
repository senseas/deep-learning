package com.deep.framework.ast.statement;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.NodeList;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.Expression;
import com.deep.framework.ast.expression.ParametersExpression;
import lombok.Data;

import java.util.List;
import java.util.Objects;

import static com.deep.framework.ast.lexer.TokenType.ELSE;
import static com.deep.framework.ast.lexer.TokenType.IF;

@Data
public class IfStatement extends Statement {
    private Expression condition;
    private NodeList thenStatement = new NodeList();
    private Statement elseStatement;
    private Statement body;
    private static IfStatement statement;

    public IfStatement(Node prarent, Expression condition, Statement body) {
        super(prarent);
        this.condition = condition;
        this.body = body;

        this.condition.setPrarent(this);
        this.body.setPrarent(this);

        getChildrens().addAll(condition, body);
    }

    public static void parser(Node node) {
        statement = null;
        Stream.of(node.getChildrens()).reduce((list, a, b) -> {
            Stream.of(a.getChildrens()).reduce((e, m, n, c) -> {
                if (m.equals(IF) && n instanceof ParametersExpression) {
                    if (b instanceof BlockStatement) {
                        a.getChildrens().removeAll(m, n);
                        statement = new IfStatement(node, (Expression) n, (BlockStatement) b);
                        node.replace(a, statement);
                        node.getChildrens().remove(b);
                        list.remove(b);
                        e.clear();
                    } else {
                        a.getChildrens().removeAll(m, n);
                        BlockStatement block = new BlockStatement(null, a.getChildrens());
                        statement = new IfStatement(node, (Expression) n, block);
                        node.replace(a, statement);
                        e.clear();
                    }
                } else if (m.equals(ELSE) && Objects.nonNull(n) && n.equals(IF)) {
                    if (b instanceof BlockStatement) {
                        a.getChildrens().removeAll(m, n);
                        IfStatement elseifStatement = new IfStatement(statement, (Expression) c, (BlockStatement) b);
                        statement.getThenStatement().add(elseifStatement);
                        node.getChildrens().removeAll(a, b);
                        list.removeAll(List.of(a, b));
                        e.clear();
                    } else {
                        a.getChildrens().removeAll(m, n);
                        BlockStatement block = new BlockStatement(null, a.getChildrens());
                        IfStatement elseifStatement = new IfStatement(statement, (Expression) c, block);
                        statement.getThenStatement().add(elseifStatement);
                        node.getChildrens().remove(a);
                        list.remove(a);
                        e.clear();
                    }
                } else if (m.equals(ELSE)) {
                    if (b instanceof BlockStatement) {
                        a.getChildrens().removeAll(m, n);
                        statement.setElseStatement((BlockStatement) b);
                        node.getChildrens().removeAll(a, b);
                        list.removeAll(List.of(a, b));
                        e.clear();
                    } else {
                        a.getChildrens().removeAll(m, n);
                        BlockStatement block = new BlockStatement(null, a.getChildrens());
                        statement.setElseStatement(block);
                        node.getChildrens().remove(a);
                        list.remove(a);
                        e.clear();
                    }
                }
            });
        });
    }
}