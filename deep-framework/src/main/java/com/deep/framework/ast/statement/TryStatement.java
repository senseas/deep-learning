package com.deep.framework.ast.statement;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.NodeList;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.Expression;
import com.deep.framework.ast.expression.ParametersExpression;
import lombok.Data;

import java.util.Objects;

import static com.deep.framework.ast.lexer.TokenType.FINALLY;
import static com.deep.framework.ast.lexer.TokenType.TRY;

@Data
public class TryStatement extends Statement {
    public Expression resources;
    private BlockStatement tryBody;
    private NodeList<CatchClause> catches;
    private BlockStatement finallyBody;
    private static TryStatement statement;

    public TryStatement(Node prarent, BlockStatement tryBody) {
        super(prarent);
        this.tryBody = tryBody;
        this.catches = new NodeList<>();

        this.tryBody.setPrarent(this);

        getChildrens().addAll(tryBody);
    }

    public TryStatement(Node prarent, Expression resources, BlockStatement tryBody) {
        super(prarent);
        this.resources = resources;
        this.tryBody = tryBody;
        this.catches = new NodeList<>();

        this.resources.setPrarent(this);
        this.tryBody.setPrarent(this);

        getChildrens().addAll(resources, tryBody);
    }

    public void addCatche(CatchClause catchClause) {
        catchClause.setPrarent(this);
        this.getCatches().add(catchClause);
        getChildrens().addAll(catchClause);
    }

    public void setFinallyBody(BlockStatement finallyBody) {
        finallyBody.setPrarent(this);
        this.finallyBody = finallyBody;
        getChildrens().addAll(finallyBody);
    }

    public static void parser(Node node) {
        CatchClause.parser(node);
        statement = null;
        Stream.of(node.getChildrens()).reduce((list, a, b, c) -> {
            Stream.of(a.getChildrens()).reduce((o, m, n) -> {
                if (m.equals(TRY) && b instanceof BlockStatement) {
                    if (n instanceof ParametersExpression) {
                        //create TryNode and set Prarent , resources , tryBody
                        statement = new TryStatement(node, (Expression) n, (BlockStatement) b);
                        //remove TryNode and Parameters
                        node.replace(a, statement);
                        node.getChildrens().remove(b);
                        list.remove(b);
                    } else {
                        //create TryNode and set Prarent , tryBody
                        statement = new TryStatement(node, (BlockStatement) b);
                        //remove TryNode and Parameters
                        node.replace(a, statement);
                        node.getChildrens().remove(b);
                        list.remove(b);
                    }
                } else if (Objects.nonNull(statement)) {
                    if (a instanceof CatchClause) {
                        statement.addCatche((CatchClause) a);
                        node.getChildrens().remove(a);
                        o.clear();
                    } else if (m.equals(FINALLY)) {
                        statement.setFinallyBody((BlockStatement) b);
                        node.getChildrens().removeAll(a, b);
                        list.remove(b);
                    }
                }
            });
        });
    }
}